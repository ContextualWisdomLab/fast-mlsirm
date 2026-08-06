# Multilevel, multiple-membership, and longitudinal design contracts

## Added

- Added a provider-neutral `fast_mlsirm.multilevel` contract namespace for
  one-hot nesting, cross-classified weighted multiple membership, and repeated
  longitudinal occasions.
- Added exact membership and occasion revision provenance, deterministic
  SHA-256 identities, 128-bit public handles, strict sum-to-one and temporal
  ordering rules, bounded resource validation, and source-text-free
  serialization.
- Added separate random-intercept/slope and stationary AR(1) state
  specifications with independently controlled lagged-response dependence.
- Added realistic contract tests, an MSA RFC, implementation plan, and APA 7th
  doctoring while reserving all psychometric estimation for future Rust cores.
