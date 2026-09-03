# Fit capability manifest

## Added

- Add `fast_mlsirm.capabilities.fit_capabilities()` and `fit_capability_manifest()` as the bounded 1.0 model-by-estimator support contract.
- Expose the same contract through `python -m fast_mlsirm.capabilities` as deterministic JSON for operator, procurement, and downstream-tool consumption without importing package internals.
- Publish `contracts/fit-capabilities-v1.json` as the canonical exact wire artifact for the emitted 1.0 manifest, so downstream consumers can compare the finite support matrix byte-for-byte without a second schema-validator dependency.
- Derive admitted estimator combinations from the validated public `FitConfig` vocabulary so the manifest cannot advertise reserved estimator identities. `BIFAC2PLM` remains marginal-only (`mmle`).
- Return fresh canonical `FitCapability` value objects and validate direct construction against the same model-estimator table, preventing caller mutation or forged frozen records from becoming later manifest authority.
- Record Rust as the production numerical owner. The manifest adds no Python statistical or psychometric arithmetic and does not change estimator behavior.
