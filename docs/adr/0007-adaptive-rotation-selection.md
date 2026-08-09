# ADR-0007 — Adaptive Rotation Selection Without a Universal Optimum

- **Status:** Accepted
- **Date:** 2026-08-09

## Context

Factor rotation criteria optimize different notions of simple/target structure and
have different objective scales. Their performance depends on loading structure,
factor correlation, cross-loadings, sample size, number of factors and target
quality. Rotation objectives are often multimodal, so a single identity start is
not evidence of the global solution.

## Decision

Rotation is implemented as:

1. an extensible Rust criterion registry with explicit value/gradient semantics;
2. shared orthogonal/oblique optimization machinery;
3. deterministic multi-start search with stationarity and basin diagnostics;
4. sign/permutation alignment and target-specific invariants;
5. a separate criterion-neutral selector based on recovery, stability,
   interpretability, degeneracy risk and theory/target evidence.

Finite multi-start output is labelled **best observed solution** rather than
`global optimum` unless global optimality is separately established.

## Invariants

- Objectives from different criteria are not directly compared as if
  commensurate.
- Target/PST weight semantics are explicit; binary-mask criteria reject arbitrary
  continuous weights rather than silently squaring/reinterpreting them.
- SPD-dependent algebra uses numerically appropriate SPD routines such as
  Cholesky/LDLᵀ rather than determinant-sign heuristics with incorrect pivot
  semantics.
- Factor sign/permutation equivalence is handled before bootstrap/recovery
  comparison.
- Bifactor general-factor columns or fixed target labels are preserved when the
  criterion requires that identity.
- GPU batched starts are added only after Rust CPU objective/gradient/stationarity
  parity evidence.

## Alternatives considered

1. Hard-code varimax as default best — rejected.
2. Expose many criteria but force users to choose without evidence — acceptable as
   a low-level API but insufficient as the product selector.
3. Registry + multi-start + criterion-neutral evidence — accepted.

## Consequences

Selection is more computationally expensive and may return a Pareto set or
indeterminate result instead of one universal winner. This is preferable to
false certainty. Policy profiles such as stability-first or theory-guided are
allowed only when their weights/ordering are explicit and versioned.

## Failure / degraded behavior

If starts fail stationarity, transforms become singular, factors collapse, or
bootstrap alignment is unstable, return diagnostics and withhold the strong
selection claim. A successful optimizer exit alone is not empirical replication.

## Verification

- analytic-gradient vs finite-difference oracles;
- covariance/pattern/structure invariants;
- deterministic multi-start across worker counts;
- singular/near-singular and target edge cases;
- bootstrap Tucker congruence after global assignment/sign alignment;
- true-loading recovery simulations across known structures;
- criterion selection frequency and false-selection evidence.

## Sources

Bernaards, C. A., & Jennrich, R. I. (2005). Gradient projection algorithms and
software for arbitrary rotation criteria in factor analysis. *Educational and
Psychological Measurement, 65*(5), 676–696.

Browne, M. W. (2001). An overview of analytic rotation in exploratory factor
analysis. *Multivariate Behavioral Research, 36*(1), 111–150.

## Supersession criteria

Supersede only if a new solver/selector provides equal criterion extensibility and
stronger recovery/globality/stability guarantees without introducing a universal-
best criterion assumption.
