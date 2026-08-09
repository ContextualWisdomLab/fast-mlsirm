# ADR-0002: Rust is the numerical authority

- Status: **Accepted**
- Date: 2026-08-09
- Owner: psychometric numerical core

## Context

The package needs high-throughput estimation while retaining independent equation-level verification. Reimplementing every numerical method separately in Python and Rust would create two scientific authorities and invite drift. At the same time, a bounded NumPy reference/fallback path is valuable for parity testing, compatibility, and diagnosis.

## Decision

Production mathematical, statistical, and psychometric arithmetic is owned by `crates/mlsirm-core`. Python owns public contracts, orchestration, validation, compatibility reference paths, and reports. `crates/fast-mlsirm-py` is the reviewed PyO3/numpy bridge.

GPU execution is a device path under the Rust backend, not an independent model or authority. CPU is the f64 reference unless a method specifies otherwise. A GPU claim requires an actual GPU kernel execution plus declared-tolerance parity evidence.

## Invariants

1. A new production estimator/kernel is implemented in Rust first.
2. Python may retain an independent formula/reference implementation for parity, but must not silently become the production owner of the same calculation.
3. Formula changes update parameterization, likelihood, analytic gradients, simulations, recovery tests, docs, and bindings together.
4. The existing MLS2PLM formula remains the accepted simple-structure specialization; full discrimination-vector MLS2PLM is a distinct complete model path.
5. CPU parallelism is coarse-grained and minimizes unnecessary context switching.
6. GPU fallback is reported as fallback, never mislabeled GPU evidence.
7. PyO3 public exports use one compatible registry/initialization design; feature PRs must not create incompatible extension-module schemes.

## Alternatives considered

- **Python-only production computation:** rejected for the package's performance and interoperability goals.
- **Independent Python and Rust production implementations:** rejected because scientific drift becomes a release risk.
- **GPU as a separate backend/model family:** rejected because device choice must not alter model semantics.

## Failure and recovery

If the Rust extension is unavailable, only explicitly supported reference/fallback behavior may run. Unsupported numerical operations fail with a clear error. Device/kernel failures follow the documented Rust device fallback policy and preserve audit evidence of which backend actually executed.

## Compatibility and rollback

Public Python signatures remain the primary compatibility surface. A Rust-internal optimization may roll back without changing serialized contracts as long as exact numerical tolerances and scientific semantics remain unchanged.

## Verification

- Rust unit/property tests and clippy/fmt.
- PyO3 build/import tests.
- Rust↔NumPy parity where a reference path exists.
- true-parameter recovery for estimators.
- explicit GPU no-skip/parity tests for GPU claims.
- allocation/resource-bound regressions for large kernels.

## Consequences

This adds binding and parity work to numerical PRs but makes performance, scientific ownership, and failure semantics explicit and auditable.
