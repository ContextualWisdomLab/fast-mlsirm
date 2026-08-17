# Governed automated-scoring validation threshold policy

## Added

- Added `ValidationPolicy` and verdict `policy_id`/`policy_version` fields so
  automated-scoring acceptance gates identify their governing policy.
- Marshaled policy thresholds (`qwk_min`, `pearson_r_min`, `degradation_max`,
  overall/subgroup SMD, `min_subgroup_n`) into the Rust `validate_scoring`
  decision owner instead of hard-coding Williamson high-stakes cutoffs only in
  Python.

## Changed

- Default policy remains `williamson_high_stakes` v1.0 with the published
  high-stakes thresholds; invalid threshold ranges fail closed before Rust work.
