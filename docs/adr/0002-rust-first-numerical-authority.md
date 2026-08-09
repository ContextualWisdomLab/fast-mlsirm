# ADR-0002: Rust-First Numerical Authority

- **Status:** accepted
- **Date:** 2026-08-09

## Context

Psychometric and mathematical correctness depends on one authoritative implementation of likelihoods, gradients, diagnostics, recovery metrics, model-selection kernels, and computationally material transforms. Duplicating production arithmetic in Python and Rust invites silent divergence, while making Python-only computation the product path conflicts with the project's performance, CPU-parallel, and GPU requirements.

## Decision

Rust is the production numerical source of truth for new mathematical and psychometric capability. Python owns public contracts, validation, orchestration, marshaling, reporting, and optional provider adapters. Retained NumPy implementations are bounded reference/fallback paths and scientific parity oracles; they are not an invitation to implement new production kernels twice.

PyO3 is the supported boundary between Python and Rust. Feature-specific binding work must converge on a coherent registry/export strategy so independently developed capabilities coexist in one installed package.

GPU is a device option of Rust-owned computation. A GPU path is promoted only after CPU/GPU numerical and recovery parity and explicit non-skip execution evidence.

## Required evidence

Depending on the method:

- equation-level Rust/reference parity;
- analytic gradient vs independent finite-difference checks;
- true-parameter bias/MAE/RMSE and interval coverage;
- response-probability/information recovery;
- factor/loading/distance recovery after correct identification alignment;
- deterministic seed and thread behavior when promised;
- CPU/GPU parity;
- bounded resource tests for large/untrusted shapes;
- PyO3 build/import/delegation tests.

Correlation alone is not parameter-recovery evidence.

## Formula-change rule

A change to a model's scientific parameterization is a model-design change, not a local performance refactor. It must update parameter shapes, simulation, likelihood/objective, gradients, estimation, identification, tests, documentation, and bindings together.

## Failure behavior

- Non-finite numerical states fail closed or return an explicit governed status.
- GPU unavailability may use the documented CPU path only where that behavior is part of the public contract; a skipped GPU test is not GPU evidence.
- Python adapters must not classify scientific state by matching unstable Rust error-message text.

## Alternatives rejected

1. **Python-first production arithmetic.** Rejected for duplicated scientific authority and insufficient performance path.
2. **GPU-specific model implementations.** Rejected because device changes must not change the scientific model.
3. **Independent ad-hoc PyO3 extension loaders per feature.** Rejected as a long-term architecture because binding/export composition becomes merge-order dependent.

## Consequences

The approach increases Rust implementation/testing work but makes scientific ownership, performance optimization, CPU/GPU parity, and downstream audit substantially clearer.
