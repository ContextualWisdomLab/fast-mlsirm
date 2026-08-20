# Bind essay validation evidence to explicit strata

## Added

- Essay validation evidence can now carry an immutable, content-addressed prompt, genre, language, automated-model-family, and rubric-version stratum so buyers can distinguish scoped evidence from intentionally pooled validation results.
- Pooled reports without an explicit stratum retain the existing Rust-backed validation metrics but add a mandatory `validation_stratification_missing` human-review trigger; mismatched rubric or automated-model identities fail closed before a stratified report is emitted.
