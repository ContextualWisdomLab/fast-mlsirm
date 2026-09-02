# Dynamic evaluation item snapshots

## Added

- Add the versioned `fast_mlsirm_dynamic_evaluation_item/v1` Published Language for evaluations whose concrete items are resolved dynamically and frozen per run without requiring a pre-existing fixed item set.
- Keep item origin, evaluation role, reference semantics, reference status, regeneration evidence, adjudication provenance, validation evidence, and linking status as independent state axes.
- Permit zero-anchor pilot and within-run evaluation while failing closed on cross-version linking without a validated anchor and immutable linking evidence.
- Treat an adjudicated reference as distinct from a validated anchor, and treat a recorded seed or generation input as provenance rather than proof of deterministic regeneration.
