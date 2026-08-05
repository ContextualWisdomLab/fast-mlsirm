# Doctoring record: enterprise issue facets calibration reports

## Decision

Add one additive enterprise orchestration function that reuses the accepted
enterprise provenance replay, shared calibration bundle, Rust-backed facets
estimator, and canonical shared calibration report objects. Do not add an
enterprise-specific report class, estimator result, score aggregation, model
selection, fairness decision, utility function, or action queue.

## Rationale

Issue #404 now has accepted source, evidence, request, observation, semantic
provider, rating-record, and calibration-bundle boundaries. The next buyer-visible
gap is a safe path from exact governed executions to provenance-bound calibration
reports without requiring callers to manually pair each criterion design with an
unbound fit.

`fit_enterprise_issue_facets_calibration_reports()` validates the descriptive
report prefix and bounded review-trigger collection before bundle assembly. It
then delegates each deterministic criterion design to the existing
`fit_scoring_facets_calibration_report()` alias. Exact object reuse preserves the
established report schema, fingerprints, error behavior, and compatibility
surface.

## Provenance and privacy invariants

Every report retains the shared design fingerprint and receives package-managed
metadata containing the exact bundle fingerprint, exact design fingerprint, and
criterion-separation marker. No source text, provider output, customer identifier,
or essay/response content is copied into the report metadata.

The function performs no provider callback and introduces no new persisted object
name. Existing descriptive opaque identifiers and task-revision fingerprints
remain authoritative.

## Numerical and scientific scope

This slice adds no statistical equation. All marginal likelihood, quadrature,
gradient, parameter update, optimization, and latent-trait arithmetic remains in
the existing Rust-backed `fit_facets` implementation. The many-facet estimator and
report-integrity equation-to-source traceability remain authoritative in the
existing automated essay calibration documentation.

A successful report is not evidence of model fit, global optimality, reliability,
validity, fairness, rater interchangeability, causal intervention value, or
high-stakes readiness. Criterion-specific reports remain separate. Enterprise
priority and expected utility require a later identified and human-validated
decision-support slice.

## Compatibility and rollback

The change is additive. Existing enterprise calibration APIs and shared report
imports remain unchanged. Rollback removes the new function, export,
documentation, tests, and changelog entry; no persisted schema migration is
required because the function returns existing shared report objects.

This compatibility objective is consistent with ISO/IEC 25010:2023 and Semantic
Versioning 2.0.0. The record does not claim standards conformity, formal ABI
certification, or release readiness.

## Verification evidence

Tests require:

- validation before bundle assembly;
- one-time bounded materialization of review-trigger generators;
- exact criterion-order delegation;
- unchanged estimator settings;
- bundle and design fingerprint binding;
- criterion-separation metadata; and
- explicit documented public exports.

Exact-head merge remains gated by Python and Rust tests, statement and branch
coverage, package and release acceptance, GPU smoke, fuzzing, Security Scan,
SAST, required approvals, and all repository protections.

## References

International Organization for Standardization. (2023). *Systems and software
engineering—Systems and software Quality Requirements and Evaluation
(SQuaRE)—Product quality model* (ISO/IEC Standard No. 25010:2023).
https://www.iso.org/standard/78176.html

Preston-Werner, T. (n.d.). *Semantic Versioning 2.0.0*. https://semver.org/
