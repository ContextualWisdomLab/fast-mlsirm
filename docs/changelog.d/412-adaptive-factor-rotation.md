### Added

- Rust-native adaptive exploratory factor rotation with a broad criterion
  registry, orthogonal and oblique gradient-projection optimization,
  deterministic multi-start search, coarse CPU multithreading, and explicit
  convergence/basin diagnostics.
- Criterion-neutral empirical selection using stability, simple structure,
  degeneracy, target recovery, bootstrap Tucker congruence, Pareto evidence,
  and declared decision policies. Objective values are never compared directly
  across criterion families or described as a proven global optimum.
- Modular PyO3 `_rotation_core` bindings and package-root Python APIs for
  criterion discovery, analytic value/gradient evaluation, multi-start
  rotation, and typed immutable solutions.
- Symmetric positive-definite Cholesky log-determinant/inverse handling for the
  Bentler criterion, including pivot-provoking and near-singular regression
  oracles.
- GPArotation-compatible complete/partial target semantics using binary
  zero-or-one masks and the loss `sum(w * residual^2)`. Continuous weights are
  available only through the separately named `lp_wls` kernel.

### Deferred

- Promax, Cubimax, iterative Lp/FSS orchestration, cluster/EIV/echelon
  procedures, user-defined compiled criteria, and a parity-verified wgpu batch
  optimizer are not part of this release slice.
