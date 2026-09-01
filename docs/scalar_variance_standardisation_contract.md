# Scalar variance standardisation contract

## Responsibility

`mlsirm-core::standardisation` owns domain-neutral static scalar covariance standardisation. For a finite scalar variance `v > 0`, standardising the covariance by its own standard deviation is the identity

`(1 / sqrt(v)) * v * (1 / sqrt(v)) = 1`.

The published contract identifier is `fast_mlsirm.scalar_variance_standardisation@1.0.0`.

The implementation validates `v` and evaluates the algebraically equivalent arithmetic ratio `v / v`, which returns exact binary64 `1.0` without hard-coding the result and without the avoidable rounding of separately evaluated square-root factors (for example `1.0000000000000002` for `v = 3`). The contract covers every finite strictly positive binary64 variance, including positive subnormal values and `f64::MAX`. Zero (including negative zero) and negative values fail because no strictly positive standard deviation exists; NaN and infinities fail separately as non-finite inputs.

## Context boundary

This primitive has no clock, temporal state equation, estimator, provider, persistence, product route or scientific-claim promotion policy. TEPP owns event/valid/assertion/document/system/available-time semantics and any temporal composition that consumes this value. A TEPP adapter may map a named ctsem quantity such as scalar `TIPREDVARstd` onto this primitive only after validating the named model-specific contract. TEPP must not retain a second production implementation after a released fast-mlsirm contract is adopted and parity is demonstrated.

The scalar contract is the one-dimensional specialization of a future matrix-compatible covariance-to-correlation contract `R = D^{-1/2} Sigma D^{-1/2}`, where `D = diag(Sigma)` and every admitted diagonal variance is finite and strictly positive. That future matrix contract must validate shape, symmetry/finite evidence and diagonal admissibility before scaling, preserve deterministic CPU `f64` reference semantics, and must not import temporal state-equation meaning. This PR intentionally implements only the scalar specialization; it does not imply that a standardized covariance matrix is the identity matrix.

## TDD evidence

The RED commit `bfb19b92c034adf68b0b38cafe7d51e831540f43` added an integration contract against a module that did not exist. The implementation was then added in `crates/mlsirm-core/src/standardisation.rs` and exposed from the crate entrypoint. Contract tests require exact `1.0` bits across the smallest positive subnormal, `f64::MIN_POSITIVE`, ordinary magnitudes, very large values and `f64::MAX`; fail-closed handling of zero/negative/non-finite input; stable public error messages; and an explicit versioned contract identity.

Hosted exact-head CI, coverage, security and review evidence remain authoritative before merge. Source-level RED lineage is not a substitute for those gates.

## Research traceability

Driver, C. C., Oud, J. H. L., & Voelkle, M. C. (2017). Continuous time structural equation modeling with R package ctsem. *Journal of Statistical Software, 77*(5), 1–35. https://doi.org/10.18637/jss.v077.i05

Driver et al. provide a concrete downstream use of covariance standardisation in ctsem reporting. fast-mlsirm owns only the reusable static arithmetic; ctsem parameter naming and temporal interpretation are not imported into this bounded context.
