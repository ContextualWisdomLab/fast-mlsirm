# Crossed multiple-membership person effects

## Added

- Added a Rust-owned MAP estimator of crossed / weighted multiple-membership person effects `u_h` (Fox & Glas, 2001; Browne, Goldstein, & Rasbash, 2001). Persons may belong to several groups at once; one-hot nesting remains the singleton special case of the same sparse design.
- Added a CPU-multithreaded Bernoulli score/information reduction and an optional wgpu GPU kernel for that hot loop, with f64 CPU fallback when no adapter is present. Sparse Newton accumulation stays on CPU. This slice does not estimate OLS or AR longitudinal states.
- Added `fast_mlsirm.multilevel.estimate_crossed_person_effects` and `CrossedPersonEffectResult` as marshal-only Python access, plus a true-parameter RMSE recovery test against simulated crossed membership weights.
- Enforced the binary-response contract before native discovery and again inside the Rust estimator: finite non-negative observed cells must be exactly `0` or `1`; negative and non-finite cells retain the established missing-data semantics.
