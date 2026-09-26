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
- **Caller-controlled bootstrap scale**: replicate count, batch size, endpoint-movement stopping ratio, and compute budget are validated caller arguments. The endpoint-movement rule is a heuristic and does not implement the Andrews–Buchinsky (2000, pp. 23–24) percentage-deviation criterion with caller-chosen `(pdb, τ)`.
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
   - Stops at batch boundaries when the maximum percentile-interval endpoint movement relative to the interval half-width falls below the caller-supplied heuristic ratio, when the compute budget is reached, or when all requested replicates complete. This early stop does not certify Monte Carlo accuracy.
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
- Does not implement the Andrews–Buchinsky `(pdb, τ)` replicate-number procedure or a Monte Carlo accuracy guarantee for early-stopped intervals.

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

## Correction (2026-09-27)

The original decision text equated successive-batch endpoint movement divided by interval half-width with Andrews and Buchinsky's percentage deviation from an ideal infinite-repetition quantity. The source defines the latter with a probability requirement `1 − τ`; the implementation observes neither the ideal quantity nor that exceedance probability. The wording above now describes the implemented rule and its limit without changing its behavior.
