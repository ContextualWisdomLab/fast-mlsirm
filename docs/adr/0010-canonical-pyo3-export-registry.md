# ADR-0010 — Canonical PyO3 and Public Export Registry

- **Status:** Accepted
- **Date:** 2026-08-09

## Context

As Rust-owned features grow, independent PRs can add bindings and package-root
exports that work alone but conflict when merged. Multiple ad-hoc secondary
PyO3 module initializers or duplicated `__init__.py` rewrites create stack-order
risk and make public API ownership difficult to audit.

## Decision

All Rust/PyO3 numerical feature surfaces converge on one maintainable binding
registration architecture and one intentional Python public-export registry.
Feature-specific binding modules are composable inputs to that registry rather
than competing initializers.

The exact implementation may evolve, but the invariant is a single integrated
registry that can be built/tested with all accepted modules present.

## Invariants

- Each Rust function/type has one public binding owner.
- New features register through the common binding architecture.
- Package-root exports are explicit and covered by tests.
- A feature PR must validate coexistence with already accepted binding modules.
- Dynamic runtime source rewriting or compile-on-import is not used to reconcile
  binding conflicts.
- Python wrappers validate/marshal and delegate; they do not duplicate the Rust
  statistical algorithm.

## Alternatives considered

1. One secondary `PyInit_*` symbol per feature — rejected as the long-term
   default because independent branches can conflict and wheel-platform behavior
   becomes difficult to reason about.
2. Separate extension wheel per small feature — rejected for fragmentation unless
   a future bounded crate merits independent distribution.
3. Common registry with modular source files — accepted.

## Consequences

Feature PRs may need small integration edits outside their immediate module.
This is preferable to merge-order-dependent binding behavior. Public API changes
remain visible in one place and package compatibility tests can enumerate the
supported surface.

## Failure / degraded behavior

If a feature cannot be integrated without changing a public name or module path,
use an explicit compatibility/deprecation migration rather than a hidden alias.
If the compiled extension is unavailable, supported reference fallbacks follow
ADR-0002; binding registry failure is not solved by Python reimplementation.

## Security and supply chain

The installed wheel contains reviewed compiled code only. Runtime does not fetch
or generate binding source. The registry must not broaden native entrypoints to
unvalidated provider payloads without a Python/Rust boundary contract.

## Verification

- cargo tests for the binding crate;
- wheel build/reinstall on supported CI platforms;
- package-root export inventory tests;
- Python→Rust delegation tests for each numerical feature;
- coexistence tests that import/call multiple registered feature families in one
  interpreter;
- ABI/import failure tests where practical.

## Supersession criteria

Supersede if bindings are split into independently versioned extension packages
with a documented compatibility graph and equal or stronger installation,
coexistence and delegation evidence.
