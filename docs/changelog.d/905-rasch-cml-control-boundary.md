# Harden Rasch CML public controls

## Fixed

- Validate Rasch CML and Andersen LR response/group inputs plus trusted iteration/tolerance controls before compiled-core discovery, rejecting caller-defined scalar coercion while preserving genuine NumPy scalar compatibility and Rust-owned conditional-likelihood arithmetic.
