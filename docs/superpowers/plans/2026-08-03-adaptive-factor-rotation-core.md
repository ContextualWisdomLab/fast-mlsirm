# Adaptive factor-rotation core implementation plan

**Goal:** Add an extensible Rust rotation registry, deterministic global multi-start diagnostics, and a criterion-neutral empirical selector without claiming a universally optimal criterion.

**Architecture:** Analytic criteria implement one loading-space value/gradient contract. Orthogonal and oblique gradient-projection optimizers are criterion-agnostic. Multi-start orchestration is coarse-threaded in Rust. A separate selector compares solutions using neutral ranks, bootstrap congruence, Pareto evidence, and explicit policy weights. Python only validates and marshals.

**Tech stack:** Rust 2021, PyO3, NumPy, pytest, cargo test, cargo llvm-cov, GitHub Actions.

---

## Task 1: Lock criterion formulas with RED tests

- Add finite-difference tests for Orthomax, Crawford–Ferguson, Oblimin, Geomin, target/PST, entropy, infomax, McCammon, Simplimax, bifactor/bi-geomin, Tandem, Oblimax, Bentler, Quartimax, Varimax/Varimin, and Lp-WLS.
- Confirm every public hyperparameter fails closed on non-finite or out-of-domain values.
- Implement formulas in `crates/mlsirm-core/src/rotation/criteria.rs`.

## Task 2: Implement manifold optimization

- Add deterministic matrix primitives and singularity checks.
- Add orthogonal projected gradients with Cayley retraction.
- Add oblique `A T^{-T}` projection and unit-column transforms.
- Add Barzilai–Borwein steps, non-monotone Armijo search, stationarity diagnostics, and failure reasons.
- Verify objective reduction and reproduced-covariance invariance.

## Task 3: Implement multi-start global-search evidence

- Generate identity plus seeded Gaussian-QR/oblique starts.
- Solve starts in fixed, coarse Rust threads with deterministic result ordering.
- Canonicalize sign/permutation equivalence while preserving target labels and the bifactor general column.
- Report best observed basin, support, distinct minima, every start value, and backend provenance.

## Task 4: Implement criterion-neutral selection

- Add neutral simple-structure, balance, degeneracy, convergence, and basin metrics.
- Add sign/permutation-aligned bootstrap Tucker congruence.
- Add optional theory-target recovery.
- Add Pareto frontier and named decision policies.
- Grade evidence as single-sample, exploratory bootstrap, or supported bootstrap.
- Prohibit direct comparison of criterion objective values.

## Task 5: Bind and document

- Expose Rust rotation and selector endpoints through PyO3.
- Add immutable Python result dataclasses and validation.
- Export public functions from `fast_mlsirm`.
- Update factor-extraction scope wording, README, changelog, and APA 7 documentation.

## Task 6: Verification gates

- Run `cargo fmt --all -- --check`.
- Run ordinary and rotation-specific Rust tests.
- Build and test the PyO3 crate.
- Run Python rotation, selection, validation, and recovery tests.
- Run 100% Python branch/docstring coverage gates.
- Run Rust coverage, Clippy, security, SAST, and repository checks on the exact PR head.
- Address every review thread, rerun checks, then merge.

## Deferred evidence-backed extensions

- Promax, Cubimax, iterative Lp/forced-simple-structure, cluster, EIV, and echelon criteria.
- User-defined compiled criterion plugin ABI.
- Batched wgpu optimizer with CPU objective/gradient/stationarity parity.
- Bootstrap extraction directly from raw data rather than caller-supplied loading matrices.
- Simulation-policy meta-learning across known population structures.
