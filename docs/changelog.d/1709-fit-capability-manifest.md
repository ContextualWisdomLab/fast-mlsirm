### Fit capability manifest

- Add `fast_mlsirm.capabilities.fit_capabilities()` and `fit_capability_manifest()` as the bounded 1.0 model-by-estimator support contract.
- Derive admitted estimator combinations from the validated public `FitConfig` vocabulary so the manifest cannot advertise reserved estimator identities. `BIFAC2PLM` remains marginal-only (`mmle`).
- Record Rust as the production numerical owner. The manifest adds no Python statistical or psychometric arithmetic and does not change estimator behavior.
