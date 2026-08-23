# Validate G-theory controls and score evidence before Rust discovery

## Fixed

- G-theory D-study sizes and `Phi(lambda)` scalar controls now fail closed before caller-owned score-array materialization and before compiled Rust capability discovery when invalid, while preserving the existing callback-free Python/NumPy scalar contract.
- `gtheory_pi()`, `gtheory_pio()`, and `phi_lambda()` now reject callback-bearing array providers, non-real storage, and complex score evidence before NumPy real narrowing or Rust discovery; ordinary exact NumPy real arrays and built-in list/tuple score trees containing concrete Python/NumPy real scalars remain supported.
- G-study ANOVA/EMS, variance-component, D-study, and `Phi(lambda)` arithmetic remain unchanged and Rust-owned.