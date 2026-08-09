### Added

- Add a factory-sealed, content-addressed post-pilot item-bank lifecycle that requires exact calibration, item-fit, DIF, information, approval, drift, suspension, and retirement evidence before an item can advance through `piloting`, `calibrated`, `approved`, `active`, `suspended`, reactivated, or terminal `retired` states.
- Preserve policy criticality independently of psychometric discrimination, require use-specific approval, link every successor to the exact previous record fingerprint, and retain only source-text-free evidence identities while leaving numerical calibration and item-bank arithmetic Rust-owned.

### Boundaries

- This contract adds no database, hosted workflow, provider SDK, new estimator, automatic approval, version bump, or release. Downstream products remain responsible for tenancy, authorization, identity mapping, persistence, encryption, retention, deletion, and human governance.
