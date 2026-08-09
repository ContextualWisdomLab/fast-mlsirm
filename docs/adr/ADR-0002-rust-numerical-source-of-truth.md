# ADR-0002 — Rust Numerical Source of Truth

- **Status:** Accepted
- **Date:** 2026-08-09
- **Decision owner:** `fast-mlsirm`
- **Implementation status:** active on protected-main numerical paths; new numerical features must conform

## Context

Psychometric estimation is computationally intensive and scientifically sensitive. Maintaining independent production implementations in Python and Rust would create silent formula drift, inconsistent edge-case behavior, duplicated review, and unclear CPU/GPU ownership.

## Decision

Production mathematical ownership resides in the Rust core. Python owns public contracts, validation, orchestration, serialization and reporting. A NumPy path may remain as an explicitly documented reference/fallback when needed for parity, not as an independently evolving production model.

CPU parallel execution uses bounded coarse-grained work and minimizes nested thread-pool/context-switch overhead. GPU acceleration is implemented as a device path for the same statistical contract and becomes accepted only after parity and no-skip evidence.

## Invariants

- Formula changes update Rust implementation, bindings, simulation/recovery, tests and documentation coherently.
- PyO3/NumPy bindings marshal data and errors; they do not reimplement the owned likelihood or estimator.
- GPU and CPU identify the same model and serialization contract.
- Resource limits are checked before expensive allocation where feasible.
- Performance changes preserve numerical/statistical behavior at declared tolerances and are benchmarked with environment metadata rather than universal speed claims.

## Alternatives considered

1. Python/NumPy as production source of truth — rejected for hot-path performance and duplicate mathematical ownership.
2. Independent Python and Rust production implementations — rejected because parity becomes a permanent dual-maintenance burden.
3. Rust source of truth with Python reference/parity path — accepted.

## Consequences

New psychometric algorithms require Rust design and tests even if Python is the user-facing API. Pure orchestration features can remain Python-only when they perform no statistical arithmetic owned by the core.

## Failure / degraded behavior

If the Rust extension is unavailable, an explicitly supported reference fallback may run only where its semantics are documented and parity-tested. A consumer requesting `backend="rust"` fails clearly rather than silently using Python. GPU fallback follows the feature's documented device policy and never fabricates GPU evidence.

## Security and privacy

Rust kernels receive already validated bounded arrays/artifacts. Untrusted parser logic remains outside unsafe/native boundaries where practical. Errors returned through PyO3 must be bounded and must not reflect sensitive payloads.

## Compatibility / rollback

Numerical-contract changes that cannot preserve public behavior require explicit version/migration policy. Rollback restores both Rust and binding/API artifacts to a single known contract; a Python-only hotfix must not mask a Rust regression.

## Verification

- Rust unit/property tests and PyO3 crate tests.
- Rust↔NumPy/reference numerical parity where a reference exists.
- True-parameter bias/RMSE/coverage and convergence evidence for new estimators.
- CPU/GPU parity and explicit no-skip evidence for GPU-owned paths.
- Wheel build/reinstall tests proving the compiled module is actually shipped.

## Sources

Kang, I., & Jeon, M. (2025). Multidimensional latent space item response models: A note on the relativity of conditional dependence. *Psychometrika, 90*(2), 799–826. https://doi.org/10.1017/psy.2025.5

Molenaar, D., & Jeon, M. (2026). Regularized joint maximum likelihood estimation of latent space item response models. *Psychometrika, 91*, 335–359. https://doi.org/10.1017/psy.2025.10068

## Supersession criteria

Supersede only if a future execution architecture demonstrates a single equally auditable numerical source of truth with stronger recovery, portability and performance evidence across supported platforms.
