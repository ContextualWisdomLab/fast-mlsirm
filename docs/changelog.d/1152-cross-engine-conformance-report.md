# Accessible cross-engine conformance evidence

## Added

- Add a deterministic standalone HTML and canonical JSON renderer for strict `ConformanceInventory` manifests, exposing capability coverage, capability × engine execution evidence, immutable inventory/run provenance, limitations, and explicit no-evidence states with exact values in text.
- Add a deterministic provenance-bound long-form JSON table so buyers can download one flat row per capability × engine evidence record without spreadsheet formula execution risk; capabilities with no independent engine remain explicit `not_executed` rows instead of disappearing or turning green.
- Escape manifest text, emit semantic table captions/headers and a restrictive no-script CSP, and state explicitly that numerical conformance is not construct validity, fairness, or high-stakes approval.
- Delegate all ingestion to strict manifest replay and keep the renderer reporting-only; no likelihood, discrepancy, RMSE/MAE, uncertainty, alignment, scoring, or other production psychometric/statistical arithmetic moves out of Rust.