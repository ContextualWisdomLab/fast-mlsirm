# ADR-0011: Converge Rust-backed features on one canonical PyO3/public-export registry

Status: **Proposed**  
Date: 2026-08-09

## Context

`fast-mlsirm` exposes Rust numerical functionality through PyO3 and composes a large Python package-root API. As independent feature PRs add Rust-backed modules (for example scoreability, rotation, future multilevel/time kernels), each can be tempted to add its own secondary `PyInit_*` symbol, loader shim, `_legacy_init.py` rewrite, or competing `python/fast_mlsirm/__init__.py` composition. Even when each PR works alone, sequential merges can produce import collisions, hidden initialization order, duplicate marshalling conventions, or one feature silently dropping another's export.

The current protected package already has working public export composition. This ADR does not declare that those existing paths are broken. It defines the direction required before future Rust feature proliferation makes them unmaintainable.

## Decision

Adopt one repository-owned **canonical binding and public-export registry architecture** for Rust-backed feature modules.

The target design shall:

1. keep `crates/mlsirm-core` as numerical authority and `crates/fast-mlsirm-py` as the single reviewed native binding crate;
2. register Rust-backed Python functions/types from feature-scoped binding modules through one explicit composition point;
3. make the Python package-root export set derive from one maintained composition layer rather than feature PRs independently rewriting `__init__.py`/legacy init state;
4. use one marshalling/error/array-ownership convention per result family and document exceptions;
5. support additive feature registration without runtime source rewriting, dynamic compilation, import-time network access, or mutable plugin discovery;
6. fail import/build tests when two features claim the same public symbol or incompatible module-initialization path; and
7. preserve backwards-compatible public imports through explicit deprecation/alias policy rather than hidden import fallback.

A secondary extension symbol may exist only as an explicitly designed module of this registry with cross-platform wheel/import evidence; it may not be invented independently by a feature PR as the easiest local integration.

## Invariants and acceptance evidence

- A wheel containing two or more Rust-backed feature families imports all of them in the same interpreter/process on every supported CI platform.
- Package-root exports include the union of intended public symbols with no order-dependent loss.
- `maturin`/PyO3 build metadata has one source of truth.
- Native exceptions are mapped through reviewed typed/bounded Python errors; Python does not parse Rust error strings to determine scientific status.
- NumPy arrays returned across the boundary have explicit ownership/immutability/shape semantics.
- An import test starts from a clean environment and never rewrites source or builds a second native module at runtime.
- Feature PRs include Rust -> PyO3 -> Python delegation tests rather than reimplementing the numerical calculation in Python.

## Consequences and trade-offs

A centralized registry creates a shared integration hotspot and can require small coordination when parallel feature PRs add bindings. That cost is preferable to multiple incompatible extension initializers and package-root compositions. Feature implementations remain modular internally; only registration/export authority is centralized.

## Alternatives considered

### Independent `PyInit_*` module per feature

Rejected as the default. It can work for isolated features but multiplies wheel/import/platform contracts and creates merge-order conflicts.

### One monolithic binding source file

Rejected. A single initializer/composition point does not require a single unmaintainable source file; feature binding modules can remain separated and registered explicitly.

### Runtime plugin discovery/dynamic compilation

Rejected for the core package. It weakens reproducibility, offline packaging, supply-chain review and release provenance.

### Python reimplementation when binding integration is difficult

Rejected for production numerical paths by ADR-0002. Binding work is part of releasing a Rust numerical feature.

## Reversal / supersession conditions

A future stable PyO3 or Python packaging mechanism that provides independently versioned subextensions with stronger reproducibility and lower integration risk may supersede this decision, but only after cross-platform wheel/import evidence and a migration plan for current public imports.

## References

PyO3 Project. (2026). *PyO3 user guide* [Software documentation]. https://pyo3.rs/

Python Software Foundation. (2026). *Extending and embedding the Python interpreter: Defining extension modules*. Python 3.14 documentation. https://docs.python.org/3.14/extending/extending.html
