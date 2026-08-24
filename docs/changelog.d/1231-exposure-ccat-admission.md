# Harden constrained-CAT evidence admission

## Fixed

- Validate CCAT ability, item, content-group, target, and administered-mask evidence before native dispatch; reject callback-bearing or lossy storage, require lossless non-negative integral `uintp` group marshalling, and leave constrained-CAT selection arithmetic Rust-owned.
