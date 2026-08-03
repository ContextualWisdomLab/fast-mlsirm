# Fail-closed Vuong selection summary

## Added

- A bounded public `compare_nonnested_models` orchestration API that preserves Rust-computed casewise likelihood-ratio mean, variance scale, corrected selection statistic, and two-sided probability together with explicit model-relation metadata.
- Auditable `ModelRelation`, `ComparisonStatus`, and immutable `ModelComparisonResult` contracts.
- Relation-appropriate routing for nested, boundary-nested, overlapping, strictly non-nested, and unknown candidate pairs.

## Safety boundary

- Omitted relation metadata defaults to `unknown`.
- The API does not report a winning model until Vuong's formal first-stage distinguishability evidence is available from a common compiled score/information contract.
- Numerical variance checks are not mislabeled as the formal weighted-chi-square distinguishability test.
- Oversized iterables, malformed labels, invalid parameter counts, and compiled-kernel rejections fail closed without leaking low-level exception text or reproducing statistical arithmetic in Python.
