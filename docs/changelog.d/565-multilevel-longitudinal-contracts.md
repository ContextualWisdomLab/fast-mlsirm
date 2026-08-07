# Multilevel, multiple-membership, and longitudinal design contracts

## Added

- Added a provider-neutral `fast_mlsirm.multilevel` contract namespace for one-hot nesting, cross-classified designs, weighted multiple membership, multiple-membership multiple-classification, and repeated longitudinal occasions. Every contextual edge names an explicit `context_dimension_id` and `context_id`; schema 1.0 never infers or invents a random-effect family from a context label.
- Added independent weight normalization within every observation-by-context-dimension group, required coverage of every declared context dimension, dimension-scoped duplicate and context identities, exact per-dimension count/weight serialization, and assignment-revision fingerprints bound to the precise observation, dimension, context, and weight.
- Added deterministic SHA-256 identities, descriptive 128-bit public handles, child-artifact replay protection, bounded and callback-safe collection handling, strict respondent-level occasion ordering, and source-text-free serialization.
- Added separate random-intercept/slope and discrete occasion-step stationary AR(1) state specifications with independently controlled lagged-response dependence. Irregular millisecond offsets remain provenance only; continuous-time or interval-adjusted transitions require a later explicit Rust contract.
- Added realistic contract and adversarial tests, an MSA RFC, staged implementation plan, and APA 7 doctoring while reserving all likelihood, integration, optimization, uncertainty, multithreading, GPU work, and true-parameter recovery for future Rust cores.
