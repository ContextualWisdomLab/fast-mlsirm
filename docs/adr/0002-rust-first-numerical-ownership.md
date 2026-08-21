# ADR-0002: Rust-first numerical ownership

Status: **Accepted**  
Date: 2026-08-09

## Context

The repository exposes Python APIs while supporting computationally intensive psychometric estimation, diagnostics, calibration and simulation. Maintaining independent Python and Rust production formulas creates drift risk, doubles verification burden and makes CPU/GPU ownership ambiguous. At the same time, explicit NumPy paths are valuable for parity, research inspection and controlled reference use.

## Decision

Rust is the production source of truth for mathematically material psychometric computation, including likelihoods, gradients, Hessians/information matrices, optimization, scoring/ranking, item/factor information and other numerical kernels when promoted to production capability.

Python may:

- validate and bound inputs;
- marshal NumPy arrays;
- orchestrate domains/providers;
- expose typed results;
- render reports;
- retain governed reference/parity calculations where a parity contract exists.

Python shall not become an independently evolving second production formula.

The public backend architecture may expose `auto`, `rust` and governed `numpy` reference/parity choices for APIs that currently support them. `auto` resolves to Rust when the compiled core is available and fails closed otherwise; it never silently selects NumPy. GPU is a Rust device path, not a third psychometric formula implementation.

The canonical PyO3 layer must support feature growth without independent PRs overwriting initialization/export structure. Secondary module symbols, if retained, must be registered through one auditable binding architecture.

## Numerical invariants

- Formula changes update Rust and any governed reference path together.
- Analytic derivatives are checked against independent finite-difference or equivalent oracles where practical.
- CPU/GPU parity uses identification-aware invariants rather than raw non-identifiable coordinates.
- Caller-controlled allocation sizes are bounded before allocation.
- Non-finite numerical boundaries fail closed where the model requires finite values.
- Computationally material CPU parallelism should minimize task/thread context-switch overhead.

## Consequences

Benefits:

- one production arithmetic authority;
- safer GPU/CPU parity and performance work;
- clearer auditability and reproducibility;
- Python remains ergonomic without owning scientific numerics.

Costs:

- new mathematical features require Rust/PyO3 work before product release;
- pure-Python prototypes cannot be declared production estimators without migration.

## Alternatives considered

1. **NumPy as primary backend, Rust as optional accelerator.** Rejected as the long-term architecture because production behavior can diverge and high-cost computations remain Python-owned.
2. **Rust-only API.** Rejected because Python remains the principal research/product integration surface.
3. **Independent CPU/GPU model implementations.** Rejected; GPU must share the same mathematical contract and parity evidence.

## Reversal conditions

A different numerical owner requires an ADR demonstrating equivalent or stronger correctness, performance, packaging, parity and scientific-evidence guarantees.
