### Fit capability manifest

- Add `fast_mlsirm.capabilities.fit_capabilities()` and `fit_capability_manifest()` as the bounded 1.0 model-by-estimator support contract.
- Expose the same contract through `python -m fast_mlsirm.capabilities` as deterministic JSON for operator, procurement, and downstream-tool consumption without importing package internals.
- Publish `contracts/fit-capabilities-v1.schema.json` as a Draft 2020-12 exact wire contract for the emitted 1.0 manifest, including the bounded model ordering and admitted estimator combinations.
- Derive admitted estimator combinations from the validated public `FitConfig` vocabulary so the manifest cannot advertise reserved estimator identities. `BIFAC2PLM` remains marginal-only (`mmle`).
- Record Rust as the production numerical owner. The manifest adds no Python statistical or psychometric arithmetic and does not change estimator behavior.
