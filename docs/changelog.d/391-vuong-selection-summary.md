# Fail-closed Vuong selection summary

## Added

- A bounded public `compare_nonnested_models` orchestration API that preserves Rust-computed casewise likelihood-ratio mean, variance scale, corrected selection statistic, and two-sided probability together with explicit model-relation metadata when the normal-selection kernel is applicable.
- Auditable `ModelRelation`, `ComparisonStatus`, and immutable `ModelComparisonResult` contracts.
- Relation-appropriate routing for nested, boundary-nested, overlapping, strictly non-nested, and unknown candidate pairs.

## Security

- Omitted relation metadata defaults to `unknown`.
- Nested, boundary-nested, and unknown relations are routed before the non-nested normal-selection kernel is invoked, so a rejected or exact-zero non-applicable statistic cannot mask the required likelihood-ratio or relation-resolution procedure.
- The API does not report a winning model until Vuong's formal first-stage distinguishability evidence is available from a common compiled score/information contract.
- Numerical variance checks are not mislabeled as the formal weighted-chi-square distinguishability test.
- Casewise inputs are bounded and normalized to finite floats before FFI; booleans, opaque values, non-finite values, conversion overflow, malformed labels, invalid parameter counts, and compiled-kernel rejections fail closed without leaking low-level exception text or reproducing statistical arithmetic in Python.
