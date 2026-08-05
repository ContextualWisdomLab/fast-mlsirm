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

## Compatibility and rollback

- Existing essay imports continue to resolve unchanged.
- New shared imports resolve to the exact same class and functions.
- Existing `essay`-prefixed handles and error codes remain stable.
- Rollback consists of removing the additive shared module and documentation; no
  persisted artifact migration is required.
- Any future wire-identity rename requires a separately versioned migration and
  dual-read compatibility evidence.

## Numerical and scientific scope

This slice adds no equation, likelihood, gradient, Hessian, optimizer, scoring,
ranking, fairness, utility, or causal computation. The existing Rust-backed
estimator and report integrity gates remain authoritative. Therefore no new
equation-to-source citation is introduced. The existing many-facet estimator
traceability remains in the automated essay calibration documentation.

## Verification evidence

The compatibility tests assert exact class and function identity, constant parity,
explicit exports, inherited public documentation, and canonical module ownership.
This prevents an accidental second report schema from entering the package.
