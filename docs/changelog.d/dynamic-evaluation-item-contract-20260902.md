# Criterion-bound dynamic evaluation snapshots

## Added

- Add the versioned `fast_mlsirm_dynamic_evaluation_item/v1` Published Language for evaluations whose concrete items are resolved dynamically and frozen per run without requiring a pre-existing fixed item set.
- Require a non-empty immutable criterion-set snapshot before either an item or a run can be admitted; zero fixed anchors remain valid for pilot and within-run evidence collection.
- Content-address criterion definitions, admissible-evidence rules, exclusion rules, response semantics, abstention rules, not-observable rules, and response-category definitions.
- Bind every admitted item and run to the exact criterion-set identity and SHA-256 digest, reject unregistered criteria or rubric/blueprint substitution, and require every declared criterion to be administered.
- Keep item origin, evaluation role, reference semantics, reference status, regeneration evidence, adjudication provenance, validation evidence, and linking status as independent state axes.
- Fail closed on cross-version linking without a validated anchor and immutable linking evidence.
- Treat an adjudicated reference as distinct from a validated anchor, and treat a recorded seed or generation input as provenance rather than proof of deterministic regeneration.
