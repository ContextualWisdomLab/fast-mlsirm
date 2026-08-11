# Rust-owned CAT ability estimation

## Changed

- Moved public CAT MLE, EAP, and ability-standard-error arithmetic from the
  Python wrapper into the compiled Rust scoring core, retaining Python only for
  validation, marshalling, and adaptive-test policy.
- Added bounded Newton MLE, prior-centred grid EAP, scoped per-dimension Rust
  workers, and device-aware information reduction with explicit handling for
  all-identical response patterns and unadministered dimensions.
- Added sentinel delegation tests, Rust edge-case tests, seeded true-trait
  recovery tests, score-equation checks, and APA 7th CAT doctoring.
