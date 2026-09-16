# ADR-0027: GPU-Parallel Polytomous Bifactor QMCEM, 2-Stage Lord-Wingersky Recursion, and 390-Replicate Joint Person Bootstrap

Status: Accepted  
Date: 2026-09-16  
Supersedes: none  
Superseded by: none

## Context

High-dimensional psychometric assessments and rubric evaluation workflows frequently utilize polytomous items organized into specific subscales and method factors alongside a general trait. For these instruments:
1. Standard unidimensional IRT fails to capture construct multidimensionality and specific group dependencies.
2. Full-information Graded Response Model (GRM) simultaneous estimation across multiple demographic/experimental groups requires stable integration over multidimensional latent spaces (e.g., G + 4 specific + 1 method factor = 6 dimensions).
3. Exact score-distribution computation for equating and testlet scoring requires efficient recursion without combinatorial explosion.
4. Robust empirical inference requires joint person bootstrap replication (390 resamples), which historically presented severe computational bottlenecks when executed sequentially.

Prior to this decision, `fast-mlsirm` supported bifactor scoreability indices and simple-structure marginal models, but lacked full-information polytomous GRM multiple-group QMCEM fitting, two-stage Lord-Wingersky score recursion, and high-throughput joint bootstrap parallelization.

## Decision drivers

- **Numerical Precision**: The two-stage Lord-Wingersky recursion must match exact direct enumeration within a strict threshold of $\le 10^{-12}$ on a 257-point grid spanning $[-8.0, 8.0]$.
- **Hardware-Parallel Throughput**: 390-replicate person bootstrap estimation must achieve $\ge 5\times$ throughput scaling on multicore hardware without Python GIL contention.
- **Reproducibility & Parity**: CPU and GPU executions, as well as deterministic replicate seeds, must produce parameter estimates matching within $10^{-6}$.
- **Methodological Scope Integrity**: Maintain simple-structure and structured bifactor contracts in dedicated modules (`bifactor_recursion` and `bifactor_grm`) without mutating existing general multidimensional contracts in place.
- **Identification & Standardization**: Enforce reference group standard normal constraints ($\mu_0 = 0, \sigma_0^2 = 1$) while freely estimating focal group distributions and computing observed information standard errors via Oakes' identity.

## Ownership and dependency direction

- **Owning Component**: `fast-mlsirm` owns the numerical kernels in `crates/mlsirm-core`, PyO3 bindings in `crates/fast-mlsirm-py`, and the typed Python orchestrators in `fast_mlsirm.polytomous_bifactor` and `fast_mlsirm.bifactor_bootstrap`.
- **Downstream Boundary**: `Psychometrics Commons` and research reporting scripts consume the exported classes (`PolytomousBifactorFit`, `BifactorBootstrapResult`) and functions as pure calculation and diagnostic APIs. No database, HTTP, or product-specific models are introduced into `fast-mlsirm`.

## Decision

We implement and verify the following components:

1. **Rust Numerical Kernels (`crates/mlsirm-core`)**:
   - `bifactor_recursion.rs`: Implements two-stage Lord-Wingersky recursion for polytomous bifactor models. Stage 1 computes within-domain score distributions conditional on the general and specific factors and integrates over the specific factor. Stage 2 convolves across independent domains conditional on the general factor.
   - `bifactor_grm.rs`: Implements Quasi-Monte Carlo EM (QMCEM) using deterministic shifted Halton sequences for joint estimation of discrimination slopes, ordered category thresholds, and multiple-group latent mean/variance vectors.
   - **Oakes Observed Information Matrix**: Implements Louis/Oakes numerical derivatives of conditional expectations to yield standard errors for all item parameters without requiring complete inversion of full Hessian matrices.
   - **Slope Bounding & Monotonicity Sensitivity**: Implements box-constrained Newton steps for discrimination parameters ($|a_{id}| \le M$), verifying monotonic non-decrease in log-likelihood across increasing bound values ($[4, 6, 8, 10]$).

2. **GIL-Free PyO3 Extension (`crates/fast-mlsirm-py`)**:
   - Exposes `fast_mlsirm._bifactor_core` initialized via `_bifactor_core_loader.py`.
   - Wraps computation-intensive EM loops and slope sweeps in `py.detach(move || { ... })`, completely releasing the Python Global Interpreter Lock during estimation.

3. **Parallel Bootstrap Dispatcher (`python/fast_mlsirm/bifactor_bootstrap.py`)**:
   - Implements stratified person bootstrap resampling preserving group proportions.
   - Dispatches replicates across `ThreadPoolExecutor` workers without process serialization overhead, scaling to 390 replicates with $\ge 70$ replicates/second throughput.
   - Calculates empirical parameter standard errors and convergence summaries.

## Invariants / acceptance evidence

1. **Recursion Precision**: On a 257-point grid $[-8.0, 8.0]$ with step $0.0625$, the maximum absolute difference between `bifactor_lord_wingersky` and `direct_enumeration_bifactor` is $\le 10^{-12}$ (`test_bifactor_lord_wingersky_matches_direct_enumeration`).
2. **Multiple-Group Identification**: Group 0 moments are strictly fixed to $\mu = 0, \sigma^2 = 1$, while focal group moments are freely estimated (`test_fit_polytomous_bifactor_multiple_group_and_oakes_se`).
3. **Oakes Standard Errors**: Standard errors for all slopes and thresholds are strictly finite, positive, and free of missing/NaN values.
4. **Monotonic Sensitivity**: Log-likelihood is monotonically non-decreasing as slope upper bounds increase across $\{4.0, 6.0, 8.0, 10.0\}$ (`test_bifactor_slope_sensitivity_monotonic_loglik`).
5. **Bootstrap Throughput & Convergence**: A 390-replicate run completes in under 6 seconds on modern multicore hardware with $\ge 95\%$ convergence and parameter parity within $10^{-6}$ (`test_390_replicate_joint_bootstrap_throughput_and_parity`).

## Non-goals and claims not made

- Does not mutate or replace the existing simple-structure MLSIRM / MLS2PLM item response models in place.
- Does not implement non-compensatory or partially-ordered multidimensional response models.
- Does not replace Hosted Psychometrics Commons assessment execution services or persistent participant schemas.

## Consequences and trade-offs

### Benefits

- High-throughput capability for large-scale replication studies and bootstrap standard error estimation.
- Exact score distributions for complex multi-domain tests without numerical instability.
- Complete type safety and memory management through Rust ownership and GIL detachment.

### Costs / risks

- High QMC draw counts ($> 10,000$) increase per-iteration CPU time; mitigated by deterministic Halton sampling and adaptive convergence tolerances.
- Multiple-group estimation requires sufficient person counts per focal group ($\ge 50$) to ensure well-conditioned group variance updates.

## Alternatives considered

- **Multiprocessing via `ProcessPoolExecutor`**: Rejected due to high memory footprint and IPC serialization cost for large response matrices. Using `py.detach` with native thread pooling achieved zero-overhead concurrency.
- **Numerical Hessian Finite Differences**: Rejected for standard errors due to $O(P^2)$ likelihood evaluation scaling; Oakes' formula leverages EM conditional expectations and converges significantly faster.

## Security and privacy implications

- Numerical operations execute strictly in-memory without disk caching or external network egress.
- Person index arrays generated during bootstrap are ephemeral and contain no participant identifiers or sensitive demographic attributes.

## Verification and release evidence

- Rust unit and integration test suite: 10/10 tests passed (`cargo test --package mlsirm-core --lib bifactor`).
- Clippy workspace audit: 0 errors across all targets with `[lints.clippy]` handling of test indexing.
- Python pytest suite: 6/6 tests passed including 390-replicate benchmark.
- CodeGraph graph index synchronized (`codegraph sync`).

## Research and standards basis

- Gibbons, R. D., & Hedeker, D. R. (1992). Full-information item bi-factor analysis. *Psychometrika*, 57(3), 423-436.
- Cai, L., Yang, J. S., & Hansen, M. (2011). Generalized full-information item bifactor analysis. *Psychological Methods*, 16(3), 221-248.
- Lord, F. M., & Wingersky, M. S. (1984). Comparison of IRT true-score and equipercentile equating. *Applied Psychological Measurement*, 8(4), 453-461.
- Oakes, D. (1999). Direct calculation of the information matrix via the EM algorithm. *Journal of the Royal Statistical Society: Series B*, 61(2), 479-482.
- Bock, R. D., & Zimowski, M. F. (1997). Multiple Group IRT. In *Handbook of Modern Item Response Theory*.

## Reversal / supersession conditions

- Discovery of numerical instability in QMCEM estimation for dimensions $> 16$ requiring full Laplace approximation or variational alternatives.
- Adoption of an alternative open standard for multidimensional item calibration across ContextualWisdomLab systems.
