# Automated-scoring assessment and policy contracts

`fast_mlsirm.scoring` begins with a provider-neutral, content-addressed
`AssessmentSpec`. The contract binds an assessment version to exact
`RubricSpecification` fingerprints and to explicit engine, calibration,
validation, adjudication, and monitoring policies before any human or automated
score observation is accepted.

This is the shared foundation for domain adapters such as essay scoring and
enterprise issue intelligence. Domain modules must extend this contract rather
than defining parallel rubric, score, engine, or audit identities.

## Construction

Use `build_assessment_spec`, not the `AssessmentSpec` constructor. The builder
accepts an explicit registry of immutable `fast_mlsirm.rubric` values and checks
that:

- every rubric fingerprint occurs exactly once in the registry;
- every rubric's `construct_id` matches one declared `ConstructSpec`;
- construct declarations cover the registry exactly without assigning a rubric
  to multiple constructs;
- the calibration policy covers every assessment rubric;
- validation, adjudication, and monitoring scopes reference only declared
  rubric, engine, and subgroup identities; and
- identifiers use descriptive two-or-more-token lower `snake_case` values.

The builder canonicalizes set-like identity collections, deep-copies and freezes
bounded JSON metadata, and exposes deterministic `canonical_json()`,
`artifact_digest()`, and `assessment_handle` values. A digest is a content
identity, not an authentication, authorization, or signature mechanism.

## Policy boundary

The first slice records policy; it does not execute scoring, calibration,
validation, adjudication, or monitoring.

- `EnginePolicy` declares permitted and required engine identities and whether
  evidence and abstention are part of the operational contract.
- `CalibrationPolicy` declares the estimator family, exact rubric scope,
  minimum item/rater support, and connected-design requirement. It does not
  implement a new estimator.
- `ValidationPolicy` declares ordered metric gates, threshold direction,
  minimum evidence count, and subgroup/rubric scope. No universal thresholds
  are supplied.
- `AdjudicationPolicy` declares transparent human-review rules without
  overwriting source observations.
- `MonitoringPolicy` declares bounded-window drift rules without silently
  selecting defaults.

Later orchestration modules must reuse existing Rust-backed estimators and
statistical functions. Python may validate, marshal, and report, but it must not
reimplement psychometric likelihoods, gradients, uncertainty, DIF, agreement,
or utility arithmetic.

## Metadata safety

Assessment metadata is untrusted configuration, not a response-content store.
Only bounded canonical JSON values are accepted. The contract rejects non-string
keys, non-finite numbers, signed-64-bit integer overflow, binary or arbitrary
objects, excessive collection width, excessive nesting, and excessive total
node count. Caller-owned mappings and arrays are recursively copied and frozen,
so mutation after construction cannot change an artifact digest.

Response text, essay content, customer complaints, source documents, and other
potentially sensitive payloads belong in separately governed evidence contracts.
They must not be inserted into assessment metadata, exceptions, digests, or
default logs.

## Scientific and operational boundary

A valid `AssessmentSpec` establishes reproducible configuration and reference
integrity only. It does not establish construct validity, scoring accuracy,
rater interchangeability, fairness, calibration connectedness, predictive
utility, causal intervention value, or readiness for high-stakes automation.
Those claims require human-anchored observations, model and parameter recovery,
held-out evaluation, subgroup and DIF evidence, drift monitoring, transparent
adjudication, and an approved decision policy.
