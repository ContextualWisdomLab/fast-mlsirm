# ADR-0027: GPU-Parallel Bifactor E-step and Joint Person Bootstrap with Caller-Controlled Stopping

Status: Accepted  
Date: 2026-09-16  
Supersedes: none  
Superseded by: none

## Context

High-dimensional psychometric assessments and rubric evaluation workflows frequently utilize polytomous items organized into specific subscales and method factors alongside a general trait. For these instruments:
1. Standard unidimensional IRT fails to capture construct multidimensionality and specific group dependencies.
2. Full-information Graded Response Model (GRM) simultaneous estimation across multiple demographic/experimental groups requires stable integration over multidimensional latent spaces (e.g., G + 4 specific + 1 method factor = 6 dimensions).
3. Exact score-distribution computation for equating and testlet scoring requires efficient recursion without combinatorial explosion.
4. Robust empirical inference requires joint person bootstrap replication, where the replicate count is a study-design decision: the library takes replicate count, batch size, Monte Carlo stopping ratio, and compute budget as caller arguments and defines no study-specific defaults.

Prior to this decision, `fast-mlsirm` supported bifactor scoreability indices and simple-structure marginal models, but lacked full-information polytomous GRM multiple-group QMCEM fitting, two-stage Lord-Wingersky score recursion, and high-throughput joint bootstrap parallelization.

## Decision drivers

- **Numerical Precision**: The two-stage Lord-Wingersky recursion must match exact direct enumeration within a strict threshold of $\le 10^{-12}$ on a 257-point grid spanning $[-8.0, 8.0]$.
- **Hardware-Parallel E-step**: The bifactor E-step runs in WGSL f32 kernels with an f64 CPU fallback; fit-level CPU/GPU agreement is asserted within a documented single-precision tolerance.
- **Caller-controlled bootstrap scale**: replicate count, batch size, stopping ratio (Monte Carlo error of interval endpoints relative to half-width, in the role of the Andrews–Buchinsky percentage-deviation bound), and compute budget are validated caller arguments; the stopping rule follows Andrews and Buchinsky (2000, §§ 2–4).
- **Reproducibility & Parity**: CPU and GPU executions, as well as deterministic replicate seeds, must produce parameter estimates matching replicate-by-replicate within device precision.
- **Methodological Scope Integrity**: Maintain simple-structure and structured bifactor contracts in dedicated modules (`bifactor_recursion` and `bifactor_grm`) without mutating existing general multidimensional contracts in place.
- **Identification & Standardization**: Enforce reference group standard normal constraints ($\mu_0 = 0, \sigma_0^2 = 1$) while freely estimating focal group distributions; empirical uncertainty comes from the joint person bootstrap.

## Ownership and dependency direction

- **Owning Component**: `fast-mlsirm` owns the numerical kernels in `crates/mlsirm-core` (`bifactor_grm`, `bifactor_recursion`, `gpu_bifactor`), PyO3 bindings in `crates/fast-mlsirm-py`, and the typed Python orchestrators in `fast_mlsirm.bifactor_grm`, `fast_mlsirm.bifactor_multigroup`, `fast_mlsirm.bifactor_recursion`, and `fast_mlsirm.bifactor_bootstrap`.
- **Downstream Boundary**: `Psychometrics Commons` and research reporting scripts consume the exported classes (`BifactorGrmFit`, `BifactorMultigroupFit`, `BifactorBootstrapResult`) and functions as pure calculation and diagnostic APIs. No database, HTTP, or product-specific models are introduced into `fast-mlsirm`.

## Decision

We implement and verify the following components:

1. **Rust Numerical Kernels (`crates/mlsirm-core`)**:
   - `bifactor_recursion.rs`: Implements two-stage Lord-Wingersky recursion for polytomous bifactor models. Stage 1 computes within-domain score distributions conditional on the general and specific factors and integrates over the specific factor. Stage 2 convolves across independent domains conditional on the general factor.
   - `bifactor_grm.rs`: Bock-Aitkin EM with Gibbons-Hedeker dimension reduction for the single-group (`fit_bifactor_grm`) and multigroup (`fit_bifactor_grm_multigroup`) polytomous bifactor GRM over caller-chosen Gauss-Hermite grids, with unconstrained slopes (reverse-keyed items representable), deterministic reflection canonicalization, deterministic multi-start selection, loud validation, and non-convergence reported via flags, never substituted.
   - `gpu_bifactor.rs`: WGSL `f32` person-parallel sweep of the reduced E-step (single-group and multigroup, including group moment accumulators), selected via the `device` field of both configs with CPU fallback. The `f64` CPU sweep remains the numerical reference.
   - **Empirical uncertainty via bootstrap**: standard errors and percentile intervals come from the joint person bootstrap (`bifactor_bootstrap.py`), not from in-fit analytic approximations.

2. **PyO3 Extension (`crates/fast-mlsirm-py`)**:
   - Exposes `fit_bifactor_grm` / `fit_bifactor_grm_multigroup` on `_core` (with a validated `device` argument) and Lord-Wingersky recursion on `_bifactor_core` via `_bifactor_core_loader.py`.

3. **Parallel Bootstrap Dispatcher (`python/fast_mlsirm/bifactor_bootstrap.py`)**:
   - Implements stratified person bootstrap resampling preserving group proportions.
   - Dispatches replicates across `ThreadPoolExecutor` workers; the Rust fit releases the GIL (`py.detach`) during estimation.
   - Stops at batch boundaries when the maximum percentile-interval endpoint movement relative to the interval half-width falls below the caller-supplied Monte Carlo stopping ratio (sequential application of Andrews & Buchinsky, 2000, §§ 2–4), when the compute budget is reached, or when all requested replicates complete.
   - Reports per-replicate convergence flags, empirical standard errors, and percentile intervals. Failed replicates are excluded, never substituted.
   - Calculates empirical parameter standard errors and convergence summaries.

## Invariants / acceptance evidence

1. **Recursion Precision**: On a 257-point grid $[-8.0, 8.0]$ with step $0.0625$, the maximum absolute difference between `bifactor_lord_wingersky` and `direct_enumeration_bifactor` is $\le 10^{-12}$ (`test_bifactor_lord_wingersky_matches_direct_enumeration`).
2. **Stage-1/2 identification and reporting**: reference-group pinning, same-seed bit-reproduction, loud validation, and non-convergence reported via flags (`tests/test_bifactor_grm.py`, `tests/test_bifactor_multigroup.py`, `tests/unit/bifactor_grm_tests.rs`).
3. **GPU/CPU E-step parity**: fit-level agreement within the documented single-precision envelope on fixtures with reverse-keyed items, single-group and multigroup (`tests/test_bifactor_gpu.py`; counts-level Rust test `estep_gpu_matches_cpu_counts_and_loglik`).
4. **Bootstrap completion & parity**: requested replicates run to completion within budget on both devices unless the caller stopping rule fires; same-seed CPU and GPU runs agree replicate-by-replicate within single-precision tolerance; per-replicate convergence is reported and failed replicates are never substituted (`tests/test_bifactor_bootstrap*.py`).

## Non-goals and claims not made

- Does not mutate or replace the existing simple-structure MLSIRM / MLS2PLM item response models in place.
- Does not implement non-compensatory or partially-ordered multidimensional response models.
- Does not replace Hosted Psychometrics Commons assessment execution services or persistent participant schemas.
- **Does not claim a library-wide distributed execution backend.** This ADR’s `device` axis is exclusive per E-step (`cpu` XOR `gpu`/`auto` with fallback). GPU detection, CPU fallback, sequential EM (GPU E-step then CPU M-step), and `ThreadPoolExecutor` replicate parallelism are **not** same-host concurrent CPU+GPU work-sharing, and **not** remote multi-host hetero execution.
- **Does not claim remote Valkey/Redis Streams transport satisfies #2001.** Optional queue transport is follow-on work; shipping Valkey only for `run_bifactor_bootstrap` does **not** close the #2001 epic.
- **Does not claim GIL release (`Python::detach`, #2000) satisfies distributed execution.** GIL detach enables same-host replicate threading; it is not proof of L3/L4 orchestration.

## Follow-on requirements — library-wide distributed execution (#2001, recorded 2026-09-19)

**Status: REQUIREMENT only — not implemented under this ADR.** Tracking: [#2001](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/2001). Corrects the narrow misread in [comment-5742140343](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/2001#issuecomment-5742140343), which scoped distributed work to optional Valkey for bootstrap only.

User-facing need: a **common execution structure** for **all public numerical APIs** (fit/EM, scoring, SE/information, diagnostics, regression contrasts, MC/bootstrap/resampling), supporting:

1. **Same-host concurrent CPU+GPU numeric split (L3):** one call partitions non-overlapping numeric shards across CPU and GPU with overlapping wall time, deterministic merge, per-shard provenance, and equivalence gates vs pure-CPU and pure-GPU references.
2. **Remote heterogeneous multi-host execution (L4):** transport-agnostic task dispatch to mixed host/arch/device pools with fail-closed cohort/version gates and per-result provenance (host, arch, lib version/sha, effective device, wall time, thread budget).
3. **Bootstrap as one consumer, not the scope:** `run_bifactor_bootstrap` is the first planned L4 consumer; other replicate-parallel families (person-fit resampling, equating bootstrap, parallel analysis, recovery MC) share the same contract.
4. **Valkey/Redis Streams as optional transport:** `XREADGROUP`/`XACK`/`XAUTOCLAIM` may drive L4 when selected; transport choice is **not** the requirement boundary.

Public API inventory, split units, and today’s L0–L2 baseline are recorded in the [#2001 inventory comment (worker `task_7d9370479d05`)](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/2001#issuecomment-5742475833). Same-host CPU+GPU code-symbol audit and A/B/C acceptance matrix: [#2001 same-host audit (worker `task_aa31e6692ad9`)](https://github.com/ContextualWisdomLab/fast-mlsirm/issues/2001#issuecomment-5742433729). Use those tables as the acceptance checklist per numerical family; do not force-parallelize sequential EM iterations, MMLE quadrature chains, finite-difference Hessians, Vuong joint likelihoods, or adaptive CAT loops without documented partition rationale.

### Epic completion gate (#2001 must **not** close when only…)

- GIL/`allow_threads` detach is merged (#2000) without L3/L4 evidence.
- ADR-0027 GPU E-step + local `ThreadPoolExecutor` bootstrap is shipped without per-family L3/L4 coverage.
- Optional Valkey for `run_bifactor_bootstrap` works without same-host concurrent split and without remote hetero provenance gates on the shared contract.
- Any single numerical family reaches L4 while others remain L0–L2 without an explicit phased rollout recorded on #2001.

Implementation evidence must be linked separately from this requirement record. Until then, treat any “distributed execution done” claim as false.

## Consequences and trade-offs

### Benefits

- High-throughput capability for large-scale replication studies and bootstrap standard error estimation.
- Exact score distributions for complex multi-domain tests without numerical instability.
- Complete type safety and memory management through Rust ownership and GIL detachment.

### Costs / risks

- Per-iteration cost grows with the caller-chosen Gauss-Hermite grid; the GPU sweep pays per-sweep buffer setup, so small problems stay CPU-faster while study-scale problems (e.g. n ≈ 1000) show multi-fold GPU speedups (measured numbers in the PR, not estimates).
- Multiple-group estimation requires sufficient person counts per focal group to ensure well-conditioned group variance updates; group sizes are caller data, not library constants.

## Alternatives considered

- **Multiprocessing via `ProcessPoolExecutor`**: Rejected due to high memory footprint and IPC serialization cost for large response matrices. Releasing the GIL around the Rust fit (`py.detach`) with native thread pooling achieves zero-overhead concurrency.
- **In-fit analytic standard errors**: not implemented at this stage; uncertainty comes from the joint person bootstrap (empirical SEs and percentile intervals from converged replicates).

## Security and privacy implications

- Numerical operations execute strictly in-memory without disk caching or external network egress.
- Person index arrays generated during bootstrap are ephemeral and contain no participant identifiers or sensitive demographic attributes.

## Verification and release evidence

- Rust unit and integration tests: bifactor suites pass (`cargo test -p mlsirm-core --features gpu bifactor`), including the counts-level E-step parity test.
- No crate-wide `erasing_op`/`identity_op` suppression; function-level `#[allow]` only where narrowly justified.
- Python pytest suites pass, including CPU/GPU E-step equivalence on Gauss-Hermite grids and the bootstrap stopping/validation/parity suite.
- Measured CPU vs GPU wall times are recorded in the PR (measured, not estimated).

## Research and standards basis

- Samejima, F. (1969). Estimation of latent ability using a response pattern of graded scores. *Psychometrika, 34*, 1–97. https://doi.org/10.1007/BF03372160
- Gibbons, R. D., & Hedeker, D. R. (1992). Full-information item bi-factor analysis. *Psychometrika*, 57(3), 423-436. https://doi.org/10.1007/BF02295430
- Cai, L., Yang, J. S., & Hansen, M. (2011). Generalized full-information item bifactor analysis. *Psychological Methods*, 16(3), 221-248. https://doi.org/10.1037/a0023350
- Lord, F. M., & Wingersky, M. S. (1984). Comparison of IRT true-score and equipercentile equating. *Applied Psychological Measurement*, 8(4), 453-461. https://doi.org/10.1177/014662168400800409
- Oakes, D. (1999). Direct calculation of the information matrix via the EM algorithm. *Journal of the Royal Statistical Society: Series B*, 61(2), 479-482. https://doi.org/10.1111/1467-9868.00188
- Jank, W. (2005). Quasi-Monte Carlo sampling to improve the efficiency of Monte Carlo EM. *Computational Statistics & Data Analysis*, 48(4), 685-701. https://doi.org/10.1016/j.csda.2004.03.019
- Andrews, D. W. K., & Buchinsky, M. (2000). A three-step method for choosing the number of bootstrap repetitions. *Econometrica, 68*(1), 23–51. https://www.jstor.org/stable/2999474 (full text: Cowles Foundation Paper No. 1001).
- Bock, R. D., & Zimowski, M. F. (1997). Multiple Group IRT. In W. J. van der Linden & R. K. Hambleton (Eds.), *Handbook of Modern Item Response Theory*. Springer.

## Reversal / supersession conditions

- Discovery of numerical instability in QMCEM estimation for dimensions $> 16$ requiring full Laplace approximation or variational alternatives.
- Adoption of an alternative open standard for multidimensional item calibration across ContextualWisdomLab systems.
