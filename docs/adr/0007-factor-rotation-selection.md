# ADR-0007: Select factor rotations with criterion-neutral evidence

- Status: **Accepted as architecture/policy**; individual criterion/backend support remains feature-version specific.
- Date: 2026-08-09
- Owner: factor-rotation/model-diagnostics layer

## Context

Varimax, Crawford–Ferguson, Oblimin, Geomin, target, bifactor, simplimax, and other rotation criteria optimize different objectives and can have multiple local minima. A single identity start or the smallest raw objective value across unlike criteria cannot establish a universally optimal rotated structure.

## Decision

Rotation is split into two layers:

1. **Within-criterion optimization:** a shared Rust orthogonal/oblique optimizer runs deterministic multi-start search and returns the best observed solution plus stationarity, basin, sign/permutation, factor-correlation, and convergence evidence.
2. **Across-criterion selection:** candidate solutions are compared using criterion-neutral evidence such as simple-structure complexity, cross-loading sparsity, factor balance/collapse, factor-correlation degeneracy, bootstrap congruence/stability, target agreement, recovery, and purpose-specific policy.

The API never claims that a finite multi-start search proves a global optimum or that one criterion is universally best.

## Invariants

1. Rotation criteria and optimizers are separate extensible Rust contracts.
2. Orthogonal and oblique transformation conventions and gradients have independent numerical-oracle tests.
3. Sign/permutation equivalence is canonicalized before stability/recovery comparison.
4. Target/PST semantics are explicit; binary masks are not silently treated as arbitrary continuous weights.
5. SPD-dependent algebra uses an SPD-safe decomposition rather than determinant-sign logic that can reject valid matrices.
6. Criterion values with different scales/definitions are not directly ranked as if commensurate.
7. Selection policy and available criteria are versioned/reportable.
8. GPU batched starts are released only after CPU/GPU objective, gradient, transform, and stationarity parity evidence.

## Alternatives considered

- **Always Varimax:** rejected because simple structure and factor correlation assumptions vary.
- **Choose the smallest raw criterion value across all rotations:** rejected because objectives are not commensurate.
- **One random/identity start:** rejected because local minima can determine the reported interpretation.
- **Call multi-start the global optimum:** rejected because finite search is empirical evidence, not proof.

## Failure and recovery

Singular transforms, collapsed factors, non-finite gradients, invalid SPD operations, or unsupported criterion parameters fail closed. When multiple solutions are near-equivalent, the result retains uncertainty/basin evidence instead of hiding it behind one deterministic label.

## Compatibility and rollback

New criteria are additive registry entries. Changing an existing criterion's mathematical definition is a breaking scientific change requiring a new version/decision and recovery evidence. Optimizer improvements may roll back if they preserve the criterion definition and output contract.

## Verification

Analytic-gradient finite-difference tests, covariance preservation, transform identities, deterministic threading/seeds, multi-start basin tests, Hungarian/sign/permutation alignment, bootstrap congruence, target-recovery simulations, and Python/Rust delegation evidence are required as applicable.

## Consequences

Rotation becomes more computationally expensive than a single closed-form/default call, but the resulting interpretation is better defended against local optima, criterion shopping, and unjustified universal claims.
