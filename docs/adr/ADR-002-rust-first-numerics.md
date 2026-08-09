# ADR-002: Rust owns production mathematical and psychometric numerics

- Status: Accepted
- Date: 2026-08-09
- Deciders: ContextualWisdomLab maintainers

## Context

The package must provide reproducible high-performance psychometric computation while remaining callable from Python. Parallel implementations can drift scientifically; Python-level vectorization can also create large temporary allocations or become a hidden production fallback that has not received the same recovery evidence.

## Decision

Production likelihoods, gradients, Hessians/information, iterative estimation, psychometric scoring/ranking, linking/equating arithmetic, model-comparison numerical kernels, and computationally material optimizers are Rust-owned.

Python may own:

- schema/contract validation;
- bounded array preparation;
- orchestration and routing;
- explicit NumPy reference implementations for parity/research/fallback where documented;
- result marshaling and reports.

CPU execution should use coarse-grained multithreading with low synchronization/context-switch overhead. Computationally material GPU execution is implemented under the same Rust statistical contract and requires explicit non-skip and parity/recovery evidence.

## Consequences

- New numerical features need Rust tests and PyO3 transport rather than Python-only production code.
- Reference implementations remain valuable scientific oracles but cannot silently redefine production semantics.
- Performance work must preserve formulas and recovery, not merely benchmark faster.
- CPU/GPU backends must agree to tolerances justified by precision and algorithm, with recovery comparisons where pointwise equality is not the right invariant.

## Alternatives considered

1. **Python/NumPy as primary production backend** — rejected for project numerical-ownership and performance/reproducibility goals.
2. **Independent CPU and GPU statistical implementations** — rejected because semantic divergence would be difficult to detect and validate.
3. **Rust without reference parity** — rejected because an independent executable reference is useful for detecting kernel/gradient mistakes.

## Evidence

Existing repository policy requires Rust↔NumPy parity for shared formulas and true-parameter recovery for scientific changes. Latent-space coordinate comparisons require invariant alignment or distance-based recovery rather than naïve coordinate equality.
