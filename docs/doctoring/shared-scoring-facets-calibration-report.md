# Doctoring record: shared scoring-facets calibration report names

## Decision

Expose domain-neutral calibration-report names from
`fast_mlsirm.scoring.calibration_reporting` by aliasing the established essay
report class and builder functions. Do not copy the schema, validation, estimator
delegation, serialization, or report arithmetic.

## Rationale

The canonical facets report is already suitable for shared criterion-specific
scoring designs, but its original module and public names are essay-specific.
Enterprise issue intelligence and future domains need a shared import path without
creating incompatible fingerprints or a parallel report contract. Exact object
identity is the narrowest change that preserves ABI and wire compatibility.

Python's import system binds imported objects into the importing module's
namespace and reuses loaded module objects through `sys.modules`. The shared
surface therefore binds new names to the existing class and functions rather than
loading, subclassing, wrapping, or reconstructing them. Tests assert object
identity directly so a future refactor cannot silently introduce a second schema
or validation path.

## Compatibility and rollback

- Existing essay imports continue to resolve unchanged.
- New shared imports resolve to the exact same class and functions.
- Existing `essay`-prefixed handles and error codes remain stable.
- Rollback consists of removing the additive shared module and documentation; no
  persisted artifact migration is required.
- Any future wire-identity rename requires a separately versioned migration and
  dual-read compatibility evidence.

The additive alias follows the compatibility objective in ISO/IEC 25010:2023 and
is consistent with Semantic Versioning 2.0.0's treatment of backward-compatible
public API additions. This record does not claim ISO conformity, formal ABI
certification, or a package release. The change remains under `Unreleased` until
a separately governed release cut.

## Numerical and scientific scope

This slice adds no equation, likelihood, gradient, Hessian, optimizer, scoring,
ranking, fairness, utility, or causal computation. The existing Rust-backed
estimator and report integrity gates remain authoritative. Therefore no new
psychometric equation-to-source citation is introduced. The existing many-facet
estimator traceability remains in the automated essay calibration documentation.

## Verification evidence

The compatibility tests assert exact class and function identity, constant parity,
explicit exports, inherited public documentation, and canonical module ownership.
This prevents an accidental second report schema from entering the package.

## References

International Organization for Standardization. (2023). *Systems and software
engineering—Systems and software Quality Requirements and Evaluation
(SQuaRE)—Product quality model* (ISO/IEC Standard No. 25010:2023).
https://www.iso.org/standard/78176.html

Preston-Werner, T. (n.d.). *Semantic Versioning 2.0.0*.
https://semver.org/

Python Software Foundation. (2026). *The import system—Python 3.14.6
documentation*. https://docs.python.org/3.14/reference/import.html
