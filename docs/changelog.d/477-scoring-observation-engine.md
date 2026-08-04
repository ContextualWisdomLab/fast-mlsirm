# Governed scoring observations and engine protocol

## Added

- The provider-neutral `fast_mlsirm.scoring` core now defines content-addressed human and automated `EngineDescriptor` values, exact `ScoringRequest` bindings, source-text-free `EvidenceReference` provenance, scored/abstained/failed/excluded `ScoreObservation` states, complete `ScoringResult` execution records, and a runtime-checkable `ScoringEngine` protocol.
- Requests bind exact assessment and rubric fingerprints, declared task families, response granularity, criterion sets, allowed rubric scores, response-content digests, and bounded content statistics without retaining raw response text. Results fail closed on missing or duplicate criterion coverage, request/engine mismatches, fabricated scores, missing terminal reasons, duplicate evidence, and mixed holistic/criterion observations.
- A deterministic offline `StaticFixtureEngine` exercises the same public contracts for tests and documentation only. The shared core adds no hosted-provider SDK, network call, credential handling, scoring inference, psychometric arithmetic, or claim of reliability, fairness, model fit, scoreability, or validity.
