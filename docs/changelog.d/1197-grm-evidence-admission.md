# Graded-response evidence admission hardening

## Fixed

- Normalize GRM integration, iteration, category, seed, and tolerance controls before caller response materialization, without invoking arbitrary scalar coercion callbacks.
- Reject complex, non-real-numeric, and infinite response storage before real-valued marshalling so observed graded-category evidence cannot be silently projected or reclassified as missing.
- Preserve the documented `NaN`/negative missingness convention, confirmatory loading validation, and Rust ownership of GRM likelihood, integration, parameter estimation, EAP, identification, and convergence arithmetic.
